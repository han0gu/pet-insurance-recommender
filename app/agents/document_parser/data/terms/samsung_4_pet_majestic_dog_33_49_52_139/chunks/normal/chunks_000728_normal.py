from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[핵연료물질]\n'
 '사용된 연료를 포함합니다. [핵연료물질에 의하여 오염된 물질] 원자핵 분열 생성물을 포함합니다.\n'
 '6. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으\n'
 '로 이용함으로써 발생한 손해'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000728',
              'chunk_char_len': 136,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
