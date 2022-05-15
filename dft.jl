### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 1ef9d536-c28a-4f48-922f-28e9c6f61d27
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 7186054e-c2f9-11ec-2494-5dc6313b4393
using Random, Distributions, FFTW, Plots, BenchmarkTools

# ╔═╡ fcae40ef-c2da-445a-ad98-8ec7fd498cc3
md"
# Processamento Digital de Sinais - 2021.2
"

# ╔═╡ 4acf9021-5dd2-4c1f-8c47-c49ce3284268
md"
Ativando o ambiente de desenvolvimento do projeto para importar os pacotes instalados:
"

# ╔═╡ 409969c2-304c-4a92-9f54-2ad98a83ecfd
plotlyjs();

# ╔═╡ 3957279f-e493-4b15-8304-a3db295932bc
md"## Implemtação DFT e iDFT"

# ╔═╡ 7f67df16-76bc-4d89-a649-01b2676ee983
md"
Começamos definindo o sinal senóidal, que será amostrado no tempo, dado por:
```math
s(t) = 0.7sin(2\pi50t) + sin(2\pi120t)
```
Além disso podemos supor que o sinal $S$ sofre uma perturbação pela ação de um ruído de amplitude $A$ que segue uma disbrituição normal:
```math
x(t) = s(t) + A n(t) \quad , \quad n(t) \sim \mathcal{N(0, 1)}
```
"

# ╔═╡ 6f22e054-4232-4ef7-8c5d-d5c3a11edb6e
begin
	𝑗 = im # 😄
	Fs=1000 # Frequência de amostragem                    
	T=1/Fs # Período de amostragem       
	L=1500 # Tamanno do sinal
	t=0:T:(L-1)*T # Vetor de tempo
	A = 2 # Amplitude do ruído
	
	S = 0.7*sin.(2π*50*t) + sin.(2π*120*t)
	X = S + A*rand(Normal(), size(S))
end;

# ╔═╡ a4560c78-2163-4e11-9a40-4b3779a5682c
md"Plotando o sinal gerado:"

# ╔═╡ 5f1aea93-ea99-4228-9165-54f76a38e39f
begin
	plot(t, X, label=nothing)
	title!("Sinal Discretizado - X[k]")
	xlabel!("k")
end

# ╔═╡ 6de62c41-e94d-496c-8df8-57b44f0e8ecd
md"
A transformada de fourier discreta do sinal $x[k]$ é dada por:

```math
X[k] = \sum_{n=0}^{N-1}{x[n] e^{-j\frac{2\pi}{N}kn}} \quad, k=0, 1, ..., N-1
```
Se definirmos $\omega_N = e^{-2\pi j/N}$ a equação anterior pode ser secrita de forma matricial como:

```math
X = \begin{bmatrix}
1 & 1 & 1 & ... & 1 \\
1 & \omega_N & \omega_N^2 & ... & \omega_N^{(n-1)} \\
1 & \omega_N^2 & \omega_N^4 & ... & \omega_N^{2(n-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega_N^{(n-1)} & \omega_N^{2(n-1)} & ... & \omega_N^{(n-1)^2}
\end{bmatrix} \begin{bmatrix}
\ x[0] \ \\
\ x[1] \ \\
\ x[2] \ \\
\vdots \\
\ x[n-1] \
\end{bmatrix} 
```

O que mostra que a DFT pode ser calculada através da multiplicacao do vetor $x$ por uma matriz densa $n \times n$, requerendo $\mathcal{O}(n^2)$ operações.
"

# ╔═╡ 93d8e4ff-36b2-4b50-ba74-c7e2b2c5f8c9
md"Implementando a DFT:"

# ╔═╡ 4723b752-a967-4537-9dd0-29af9ca5d879
function dft(x)
    N = length(x)
    Ω = Array{ComplexF64}(undef, N, N)
	ωₙ = ℯ^(-𝑗*(2π/N))
    for i ∈ 1:N
		for j ∈ 1:N
			Ω[i, j] = ωₙ^((i-1)*(j-1))
		end
    end
    return Ω * x
end

# ╔═╡ 4f368f95-acca-43e6-aeed-363366a68e65
md"Em seguida, calculamos a DFT para o sinal amostrado e comparamos com a implemtação disponível na biblioteca FFTW para validar o resultado:"

# ╔═╡ 58f41eb0-4cf3-4422-91d7-a915790baf28
md"
Para recuperar o vetor do sinal discretizado, utilizamos a transformada inversa de fourier discreta, dada por:

```math
x[k] = \frac{1}{N} \sum_{n=0}^{N-1}{X[n] e^{j\frac{2\pi}{N}kn}} \quad, k=0, 1, ..., N-1
```
"

# ╔═╡ 11eb6a9e-b299-433a-8eb5-4d761c1293b1
function idft(x)
    N = length(x)
    Ω = Array{ComplexF64}(undef, N, N)
	ωₙ = ℯ^(𝑗*(2π/N))
    for i ∈ 1:N
		for j ∈ 1:N
			Ω[i, j] = ωₙ^((i-1)*(j-1))
		end
    end
    return (Ω * x) ./ N
end

# ╔═╡ 889aba5e-fb3d-428f-ac12-d971f064fdc9
md"Validando a implementação:"

# ╔═╡ 1921c4f1-4610-437b-bb65-4a0d3dae98c6
begin
	Y = dft(X)
	_X = idft(Y)
	
	plot(t, real.(_X), label="Sinal Recuperado")
	plot!(t, X, label="Sinal Original")
	title!("Recuperação do Sinal - iDFT")
	xlabel!("k")
end

# ╔═╡ fff3dc14-c2d3-4ab9-82d4-2e52ae4e4230
md"
## Implementando um filtro simples

Se desejamos filtrar, digamos apenas a senoide de 50Hz podemos excluir com uma máscara o intervalo que contém amplitudes maiores que 0.7 no grafico do modulo do espectro e recuperar o sinal:
"

# ╔═╡ 8ae358e7-dd5a-42ec-91fb-653261cb95ab
begin
	ϵ = 0.001
	index = findall(
		function compare(value)
			value >= 0.5 - ϵ
		end, Y₂
	)
	mask = zeros(L)
	mask[index] .= 1.
	plot(f, Y₂ .* mask, label="DFT Filtrada")
	title!("Amplitude da DFT Filtrada - X[k]")
	xlabel!("f (Hz)")
	ylabel!("|X[k]|")
end

# ╔═╡ 5517b706-a280-4ea2-b0e4-fd1f91ec240b
md"
Recuperando o sinal filtrado:
"

# ╔═╡ e1b06737-47b9-4caf-b1e3-1d79054bccbb
begin
	Y_f = mask .* dft(X)
	plot(t, real.(idft(Y_f)), label="Sinal Filtrado")
	plot!(t, X, label="Sinal Original")
	plot!(t, S, label="Sinal Original - Sem Ruído")
	title!("Filtragem do Sinal")
	xlabel!("k")
end

# ╔═╡ 76da28ea-cbe3-40a2-84c3-7ef63de5fd78
md"
## Implementação da FFT - Radix 2 e Radix 3
"

# ╔═╡ 097b7252-2d08-41b0-9d23-c58408744e78
md"
Apesar de simples, a implementação da DFT é custosa quando computada matricialmente como podemos observar comparando a execução com a FFT para a mesma entrada:
"

# ╔═╡ 43d248c0-48ec-472b-b6dc-4fdeadc0e759
time_dft = @elapsed begin
	dft(X)
end

# ╔═╡ 22dbfc3e-dbfb-4904-a5f2-d721c68960f3
time_fft = @elapsed begin
	fft(X)
end

# ╔═╡ 0a19230c-10ec-401b-8ac0-4eaed8f67181
md"Calculando a razão entre os tempos, podemos perceber o quão a FFT chega a ser mais eficiente:"

# ╔═╡ 64a738cb-8c22-418a-b9f5-587481f88466
time_dft / time_fft

# ╔═╡ 825e0b02-fde9-4e27-8153-e29e71405412
md"Para vetores X de tamanho crescente também podemos visualizar as curvas de desempenho:"

# ╔═╡ 6c9fb943-04f7-43b9-bbff-03e1c680b697
begin
	sizes = 1:10:100
	function pad_zeros(x, b=2)
		N = length(x)
		pad = b^Int(ceil(log(b, N)))
		return vcat(x,zeros(pad - N))
	end
	function elapsed_time(size, foo, pad=nothing)
		if pad != nothing
			x = pad_zeros(rand(size), pad)
		else
			x = rand(size)
		end
		dt = @elapsed begin
			foo(x)
		end
	end
	times_fft = [elapsed_time(Int(s), fft) for s in sizes]
	times_dft = [elapsed_time(Int(s), dft) for s in sizes]
	plot(sizes, times_dft, label="DFT - O(n*n)")
	plot!(sizes, times_fft, label="FFT - O(n*log2 n)")
end

# ╔═╡ 18e42268-d03e-420a-aff6-2fcbcfe4759e
function plot_magnitude(X, foo, foo_label, base=nothing)
	
	if base != nothing
		X = pad_zeros(X, base)
	end

	L = length(X)
	
	Y₁ = dft_amp(X, fft)
	Y₂ = dft_amp(X, foo)
	
	f = Fs*(0:L-1)/L;
	plot(f, Y₁, label="FFTW")
	plot!(f, Y₂, label=foo_label)
	title!("Espectro Frequencial")
	xlabel!("f (Hz)")
	ylabel!("|X[k]|")
end

# ╔═╡ eea859de-8c2a-4b39-b12b-01e5fe38309a
plot_magnitude(X, dft, "DFT")

# ╔═╡ 2f24447f-398e-43ae-a69e-e33e239b1a00
md"
Para implementar a transformada de fourier rápida com decimação no tempo (Radix 2 FFT) observamos que a expressão da DFT pode ser reescrita da seguinte maneira:

```math
X[k] = E_k + e^{-j\frac{2\pi}{N}k} O_k 
```

```math
X[k + \frac{N}{2}] = E_k - e^{-j\frac{2\pi}{N}k} O_k
```

Onde $E_k$ e $O_k$ são DFTs de tamanho $N/2$ relativas aos indices pares e ímpares do vetor original $x$.
"

# ╔═╡ 0175269c-25e9-43d3-8cfe-e4cae367e2a7
function dit2fft(x)
	N = length(x)

	if N == 1
		return x
	else
		odd_index = Int.(1:2:N)
		X = Array{ComplexF64}(undef, N)
		X[1:N÷2] = dit2fft(x[odd_index .+ 1])
		X[N÷2+1:N] = dit2fft(x[odd_index])
	end

	for k ∈ 0:(N÷2)-1
		p = X[k+1]
		q = ℯ^(-𝑗*(2π/N)*k)*X[k+1 + (N÷2)]
		X[k+1] = p + q
		X[k+1 + (N÷2)] = p - q
	end

	return X
end

# ╔═╡ 4006f29e-c9aa-4d0a-bbd5-ef7a1e13711f
md"
De maneira similar, podemos mostrar que para a transformada rápida de fourier (Radix 3 FFT) podemos escrever a DFT da seguinte maneira:

```math
X[k] = A_k + \omega_N^k B_k + \omega_N^{2k} C_k
```

```math
X[k + \frac{N}{3}] =  A_k + e^{-j\frac{2\pi}{3}} \omega_N^k B_k + e^{-j\frac{4\pi}{3}} \omega_N^{2k} C_k
```

```math
X[k + \frac{2N}{3}] = A_k + e^{-j\frac{4\pi}{3}} \omega_N^k B_k + e^{-j\frac{2\pi}{3}} \omega_N^{2k} C_k
```

Onde $A_k$, $B_k$ e $C_k$ são DFTs de tamanho $N/3$.
"

# ╔═╡ bcabac6a-7cb6-459c-985a-bc9084d91f4c
begin
	ω(n, N) = ℯ^(-𝑗*(2π/N)*n)
	function dit3fft(x)
		N = length(x)
	
		if N == 1
			return x
		else
			index = Int.(1:3:N)
			X = Array{ComplexF64}(undef, N)
			X[1:N÷3] = dit3fft(x[index])
			X[N÷3+1:2*(N÷3)] = dit3fft(x[index .+ 1])
			X[2*(N÷3)+1:N] = dit3fft(x[index .+ 2])
		end
	
		for k ∈ 0:(N÷3)-1
			A = X[k+1]
			B = ω(k, N) * X[k+1 + (N÷3)]
			C = ω(2k, N) * X[k+1 + 2*(N÷3)]
			X[k+1] = A + B + C
			X[k+1 + (N÷3)] = A + ω(1, 3) * B + ω(2, 3) * C
			X[k+1 + 2*(N÷3)] = A + ω(2, 3) * B + ω(1, 3) * C
		end
	
		return X
	end
end

# ╔═╡ da6e3b58-727f-4afd-82a3-a80596221a09
md"Validando mais uma vez a implementação:"

# ╔═╡ 0686643c-e5c3-4cfd-a0a5-796697634217
plot_magnitude(X, dit2fft, "Radix 2 - FFT", 2)

# ╔═╡ 5ab9ce40-055f-4d77-aea2-2acc66c7bf21
plot_magnitude(X, dit3fft, "Radix 3 - FFT", 3)

# ╔═╡ 00062d0c-b996-4b3e-a52e-86afa7c0e51e
md"Comparando as curvas de desempenho:"

# ╔═╡ 9ab655f9-602a-45ac-ae3c-e56534a6e5f6
begin
	times_dit2fft = [elapsed_time(Int(s), dit2fft, 2) for s in 1:10:100]
	times_dit3fft = [elapsed_time(Int(s), dit3fft, 3) for s in 1:10:100]
	plot(sizes, times_dft, label="DFT - O(n*n)")
	plot!(sizes, times_fft, label="FFT - O(n*log2 n)")
	plot!(sizes, times_dit2fft, label="Radix-2 FFT - O(n*log2 n)")
	plot!(sizes, times_dit3fft, label="Radix-3 FFT - O(n*log3 n)")
end

# ╔═╡ d19dea2c-2235-4278-b043-283eeec6e86c
md"
## Atividade 01 - AB1

1. Considere a sequência $x[n] = [6 \ 8 \ 5 \ 4 \ 5 \ 6]$. Implemente o algoritmo da **Transformada de Fourier Discreta (DFT)**, para $6$, $8$ e $32$ pontos e analise o espectro frequencial desse sinal, validando os resultados com uma função `fft` já implementada. Implemente também a **Transformada Discreta Inversa de Fourier (IDFT)** para restaurar a sequência original.

2. Implemente o algoritmo de **raiz de 2 (Radix-2)** e de **raiz de 3 (Radix-3)**, com decimação no tempo, da **Transformada Rápida de Fourier** (FFT) para analisar o espectro frequencial do sinal da Atividade 1. Valide os resultados com uma função `fft` já implementada.
"

# ╔═╡ 5c8b1dd5-d937-406e-a40a-8a80f2777232
x = [6, 8, 5, 4, 5, 6]

# ╔═╡ cef3ac08-7aea-4e0c-89c1-628b255cb619
begin
	xL = length(x)
	x_fft = dft_amp(x, fft)
	x_dft = dft_amp(x, dft)
	
	xf = Fs*(0:xL-1)/xL;
	plot(xf, x_fft, label="FFTW")
	plot!(xf, x_dft, label="DFT")
	title!("Amplitude da DFT - X[k] tamanho 6")
	xlabel!("f (Hz)")
	ylabel!("|X[k]|")
end

# ╔═╡ 396d6af5-87ce-480b-949d-c98021806607
begin
	x8 = pad_zeros(x)
	x8L = length(x8)
	x8_fft = dft_amp(x8, fft)
	x8_dft = dft_amp(x8, dft)
	
	x8f = Fs*(0:x8L-1)/x8L;
	plot(x8f, x8_fft, label="FFTW")
	plot!(x8f, x8_dft, label="DFT")
	title!("Amplitude da DFT - X[k] tamanho 8")
	xlabel!("f (Hz)")
	ylabel!("|X[k]|")
end

# ╔═╡ 21be5ff7-a5e0-4aef-8361-2e8704c5856a
begin
	x16 = pad_zeros(vcat(x8, [0]))
	x32 = pad_zeros(vcat(x16, [0]))
	x32L = length(x32)
	x32_fft = dft_amp(x32, fft)
	x32_dft = dft_amp(x32, dft)
	
	x32f = Fs*(0:x32L-1)/x32L;
	plot(x32f, x32_fft, label="FFTW")
	plot!(x32f, x32_dft, label="DFT")
	title!("Amplitude da DFT - X[k] tamanho 32")
	xlabel!("f (Hz)")
	ylabel!("|X[k]|")
end

# ╔═╡ b57c6915-a6db-499f-a655-0598398f5227
md"
### Referências
Kutz, J. N., Brunton, S. L. (2019). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Singapore: Cambridge University Press.

Rao, K., Kim, D., & Hwang, J. (2011). Fast Fourier Transform - Algorithms and Applications. Springer Netherlands.
"

# ╔═╡ 6d481e26-c7fe-46cc-b4f7-9703d40dd3e0
md"
### Links
* [Fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
* [Cooley–Tukey FFT algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
* [Exemplo FFT - MATLAB](https://www.mathworks.com/help/matlab/ref/fft.html)
* [FFTW](https://www.fftw.org/)
"

# ╔═╡ Cell order:
# ╟─fcae40ef-c2da-445a-ad98-8ec7fd498cc3
# ╟─4acf9021-5dd2-4c1f-8c47-c49ce3284268
# ╠═1ef9d536-c28a-4f48-922f-28e9c6f61d27
# ╠═7186054e-c2f9-11ec-2494-5dc6313b4393
# ╠═409969c2-304c-4a92-9f54-2ad98a83ecfd
# ╟─3957279f-e493-4b15-8304-a3db295932bc
# ╟─7f67df16-76bc-4d89-a649-01b2676ee983
# ╠═6f22e054-4232-4ef7-8c5d-d5c3a11edb6e
# ╟─a4560c78-2163-4e11-9a40-4b3779a5682c
# ╟─5f1aea93-ea99-4228-9165-54f76a38e39f
# ╟─6de62c41-e94d-496c-8df8-57b44f0e8ecd
# ╟─93d8e4ff-36b2-4b50-ba74-c7e2b2c5f8c9
# ╠═4723b752-a967-4537-9dd0-29af9ca5d879
# ╟─4f368f95-acca-43e6-aeed-363366a68e65
# ╠═18e42268-d03e-420a-aff6-2fcbcfe4759e
# ╠═eea859de-8c2a-4b39-b12b-01e5fe38309a
# ╟─58f41eb0-4cf3-4422-91d7-a915790baf28
# ╠═11eb6a9e-b299-433a-8eb5-4d761c1293b1
# ╟─889aba5e-fb3d-428f-ac12-d971f064fdc9
# ╠═1921c4f1-4610-437b-bb65-4a0d3dae98c6
# ╟─fff3dc14-c2d3-4ab9-82d4-2e52ae4e4230
# ╠═8ae358e7-dd5a-42ec-91fb-653261cb95ab
# ╟─5517b706-a280-4ea2-b0e4-fd1f91ec240b
# ╠═e1b06737-47b9-4caf-b1e3-1d79054bccbb
# ╟─76da28ea-cbe3-40a2-84c3-7ef63de5fd78
# ╟─097b7252-2d08-41b0-9d23-c58408744e78
# ╠═43d248c0-48ec-472b-b6dc-4fdeadc0e759
# ╠═22dbfc3e-dbfb-4904-a5f2-d721c68960f3
# ╟─0a19230c-10ec-401b-8ac0-4eaed8f67181
# ╠═64a738cb-8c22-418a-b9f5-587481f88466
# ╟─825e0b02-fde9-4e27-8153-e29e71405412
# ╠═6c9fb943-04f7-43b9-bbff-03e1c680b697
# ╟─2f24447f-398e-43ae-a69e-e33e239b1a00
# ╠═0175269c-25e9-43d3-8cfe-e4cae367e2a7
# ╟─4006f29e-c9aa-4d0a-bbd5-ef7a1e13711f
# ╠═bcabac6a-7cb6-459c-985a-bc9084d91f4c
# ╟─da6e3b58-727f-4afd-82a3-a80596221a09
# ╠═0686643c-e5c3-4cfd-a0a5-796697634217
# ╠═5ab9ce40-055f-4d77-aea2-2acc66c7bf21
# ╟─00062d0c-b996-4b3e-a52e-86afa7c0e51e
# ╠═9ab655f9-602a-45ac-ae3c-e56534a6e5f6
# ╟─d19dea2c-2235-4278-b043-283eeec6e86c
# ╠═5c8b1dd5-d937-406e-a40a-8a80f2777232
# ╠═cef3ac08-7aea-4e0c-89c1-628b255cb619
# ╠═396d6af5-87ce-480b-949d-c98021806607
# ╠═21be5ff7-a5e0-4aef-8361-2e8704c5856a
# ╟─b57c6915-a6db-499f-a655-0598398f5227
# ╟─6d481e26-c7fe-46cc-b4f7-9703d40dd3e0
