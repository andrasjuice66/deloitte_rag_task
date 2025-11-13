"""Quick test script to verify local Hugging Face setup."""

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ torch imported successfully")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        import transformers
        print("✓ transformers imported successfully")
        print(f"  Version: {transformers.__version__}")
        
        from langchain_community.llms import HuggingFacePipeline
        print("✓ HuggingFacePipeline imported successfully")
        
        from src.config import Config
        print("✓ Config imported successfully")
        print(f"  LLM Model: {Config.LLM_MODEL}")
        print(f"  Use Local LLM: {Config.USE_LOCAL_LLM}")
        print(f"  Web Search Enabled: {Config.ENABLE_WEB_SEARCH}")
        
        from src.agent import GloomhavenAgent
        print("✓ GloomhavenAgent imported successfully")
        
        from src.main import GloomhavenRulebookSystem
        print("✓ GloomhavenRulebookSystem imported successfully")
        
        print("\n✓ All imports successful!")
        print("\n📋 Configuration:")
        print(f"   Model: {Config.LLM_MODEL}")
        print(f"   Max Length: {Config.LLM_MAX_LENGTH}")
        print(f"   Temperature: {Config.LLM_TEMPERATURE}")
        print(f"   Web Search: {'Enabled' if Config.ENABLE_WEB_SEARCH else 'Disabled'}")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_model_availability():
    """Test if the model can be accessed from Hugging Face."""
    print("\n" + "="*70)
    print("Testing model availability...")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer
        from src.config import Config
        
        print(f"\nChecking if {Config.LLM_MODEL} is accessible...")
        print("(This will not download the full model, just check availability)")
        
        # This will check if the model exists but won't download the full model
        tokenizer = AutoTokenizer.from_pretrained(
            Config.LLM_MODEL,
            trust_remote_code=True
        )
        
        print(f"✓ Model {Config.LLM_MODEL} is accessible!")
        print(f"  Tokenizer vocabulary size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Could not access model: {e}")
        print("\nNote: This could be due to:")
        print("  1. Network connectivity issues")
        print("  2. Invalid model name")
        print("  3. Hugging Face Hub access issues")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Local Hugging Face Setup - Verification Test")
    print("="*70)
    print()
    
    # Test imports
    if not test_imports():
        return
    
    # Test model availability (optional)
    print("\n" + "="*70)
    response = input("Would you like to test model availability? (y/n): ")
    
    if response.lower().strip() == 'y':
        test_model_availability()
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 'python example.py' to see the full system in action")
    print("2. The model will be downloaded on first use (~2.7 GB)")
    print("3. See LOCAL_MODEL_SETUP.md for more details")


if __name__ == "__main__":
    main()




