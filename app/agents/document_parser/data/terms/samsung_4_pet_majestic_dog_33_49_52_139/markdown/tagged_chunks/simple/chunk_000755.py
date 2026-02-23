from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항에도 불구하고 보험계약에서 지정대리청구인의 지정 기간을 별도로 제한한 경\n'
 '- 우, 계약자는 이 특별약관에서도 그 기간에 한하여 지정대리청구인을 변경 지정할 수\n'
 '- 있습니다.\n'
 '- 제5조 (보험금 지급 등의 절차)\n'
 '- ① 지정대리청구인은 제6조(보험금의 청구)에 정한 구비서류 및 제1조(적용대상)의 보험\n'
 '- 수익자가 보험금을 직접 청구할 수 없는 특별한 사정이 있음을 증명하는 서류를 제출\n'
 '- 하고 회사의 승낙을 얻어 제1조(적용대상)의 보험수익자의 대리인으로서 보험금(사망'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000755',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
