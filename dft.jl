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
	L=500 # Tamanno do sinal
	t=0:T:(L-1)*T # Vetor de tempo
	A = 0. # Amplitude do ruído
	
	S = 0.7*sin.(2π*50*t) + sin.(2π*120*t)
	X = S + A*rand(Normal(), size(S)); # Desconsiderar ruído, por enquanto
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

# ╔═╡ eeee3f09-1c1b-4a85-9dce-1bcd6f3f9814
begin
	function dft_amp(X, foo)
		Y = foo(X)
		return 2*abs.(Y/L)
	end
	Y₁ = dft_amp(X, fft)
	Y₂ = dft_amp(X, dft)
end;

# ╔═╡ ad34d021-99f1-41c5-9bba-21986944b364
begin
	f = Fs*(0:L-1)/L;
	plot(f, Y₁, label="FFTW")
	plot!(f, Y₂, label="dft")
	title!("Amplitude da DFT - X[k]")
	xlabel!("f (Hz)")
	ylabel!("|X[k]|")
end

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
	index = findfirst(
		function compare(value)
			eps = 0.001
			0.7 - eps <= value <= 0.7 + eps
		end, Y₂
	)
	Y₂[index+1:L-index] .= 0.
	plot(f, Y₂, label="DFT Filtrada")
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
	Y_f = dft(X)
	Y_f[index+1:L-index] .= 0.	
	plot(t, real.(idft(Y_f)), label="Sinal Filtrado")
	plot!(t, X, label="Sinal Original")
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
	function elapsed_time(size, foo)
		dt = @elapsed begin
			foo(rand(size))
		end
	end
	times_fft = [elapsed_time(Int(s), fft) for s in 1:10:200]
	times_dft = [elapsed_time(Int(s), dft) for s in 1:10:200]
	plot(times_dft, label="DFT - O(n*n)")
	plot!(times_fft, label="FFT - O(n*log n)")
end

# ╔═╡ d19dea2c-2235-4278-b043-283eeec6e86c
md"
## Atividade 01 - AB1

1. Considere a sequência $x[n] = [6 \ 8 \ 5 \ 4 \ 5 \ 6]$. Implemente o algoritmo da **Transformada de Fourier Discreta (DFT)**, para $6$, $8$ e $32$ pontos e analise o espectro frequencial desse sinal, validando os resultados com uma função `fft` já implementada. Implemente também a **Transformada Discreta Inversa de Fourier (IDFT)** para restaurar a sequência original.

2. Implemente o algoritmo de **raiz de 2 (Radix-2)** e de **raiz de 3 (Radix-3)**, com decimação no tempo, da **Transformada Rápida de Fourier** (FFT) para analisar o espectro frequencial do sinal da Atividade 1. Valide os resultados com uma função `fft` já implementada.
"

# ╔═╡ b57c6915-a6db-499f-a655-0598398f5227
md"
### Referências
Kutz, J. N., Brunton, S. L. (2019). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Singapore: Cambridge University Press.
"

# ╔═╡ 6d481e26-c7fe-46cc-b4f7-9703d40dd3e0
md"
### Links úteis
* [FFTW](https://www.fftw.org/)
* [Implementação Matlab fft](https://www.mathworks.com/help/matlab/ref/fft.html)
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
# ╠═eeee3f09-1c1b-4a85-9dce-1bcd6f3f9814
# ╠═ad34d021-99f1-41c5-9bba-21986944b364
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
# ╟─d19dea2c-2235-4278-b043-283eeec6e86c
# ╟─b57c6915-a6db-499f-a655-0598398f5227
# ╟─6d481e26-c7fe-46cc-b4f7-9703d40dd3e0
