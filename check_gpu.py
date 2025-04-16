import torch
import platform
import multiprocessing
import subprocess

def check_gpu():
    # Verificar hardware gráfico disponível
    try:
        gpu_info = subprocess.check_output('lspci | grep -E "VGA|3D|Display"', shell=True).decode('utf-8').strip()
        has_nvidia_gpu = "NVIDIA" in gpu_info
    except:
        has_nvidia_gpu = False

    # Verificar se existe GPU disponível para acelerar o processamento
    has_cuda = torch.cuda.is_available()

    # Otimizações para sistemas sem GPU CUDA
    if not has_cuda:
        # Configurar número de threads para otimizar CPU
        try:
            # Obter número de CPUs lógicas disponíveis
            num_cpus = multiprocessing.cpu_count()
            # Configurar PyTorch para usar todos os núcleos
            torch.set_num_threads(num_cpus)
            print(f"Otimização: PyTorch configurado para usar {torch.get_num_threads()} threads de CPU")
        except Exception as e:
            print(f"Erro ao configurar otimizações de CPU: {e}")

    # Configuração de hardware
    print(f"\n===== CONFIGURAÇÃO DE EXECUÇÃO =====")
    print(f"Sistema: {platform.system()} {platform.release()}")
    print(f"Processador: {platform.processor()}")
    print(f"Núcleos CPU: {multiprocessing.cpu_count()}")

    if has_nvidia_gpu:
        print(f"GPU NVIDIA detectada no hardware")
        if has_cuda:
            print(f"CUDA disponível: Utilizando GPU para processamento")
            print(f"Modelo: {torch.cuda.get_device_name(0)}")
        else:
            print(f"⚠️ AVISO: GPU NVIDIA detectada, mas CUDA não está disponível!")
            print(f"Para melhor desempenho, instale os drivers NVIDIA e CUDA:")
            print(f"  - Em Arch Linux: sudo pacman -S nvidia cuda")
            print(f"  - Após instalar, reinicie o sistema.")
    else:
        print(f"Nenhuma GPU NVIDIA detectada no hardware")
        print(f"Usando CPU para processamento")

    print(f"====================================\n")

