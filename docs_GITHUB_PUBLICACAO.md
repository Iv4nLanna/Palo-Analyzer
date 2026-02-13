# Como publicar no GitHub

## 1) Criar repositório remoto
- Crie um repositorio vazio no GitHub (ex.: `analyzer-palo`).
- Nao marque opcao de README no GitHub (ja existe localmente).

## 2) Conectar e enviar
No terminal, dentro da pasta do projeto:

```powershell
git init
git add .
git commit -m "feat: initial release - analyzer palo"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/analyzer-palo.git
git push -u origin main
```

## 3) Atualizacoes futuras
```powershell
git add .
git commit -m "feat: descricao da mudanca"
git push
```

## 4) Recomendacoes
- Nao subir dados sensiveis (imagens de avaliados, laudos identificados).
- Mantenha modelos grandes fora do repo, se necessario use Releases ou storage externo.
