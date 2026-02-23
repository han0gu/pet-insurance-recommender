from langchain_core.documents import Document

chunk = Document(
    page_content=('사이에 보험증권에 기재된 반려견의 상해 또는 질병으로 인한 위험을 보장하기 위하여\n'
 '체결됩니다.# 제2조 (용어의 정의)이 특별약관에서 사용되는 용어의 정의는 이 특별약관의 다른 조항에서 달리 정의되지\n'
 '않는 한 다음과 같습니다.# ① 계약 관련 용어- 1. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다.\n'
 '- 2. 보험수익자: 보험금 지급사유가 발생하는 때에 회사에 보험금을 청구하여 받을 수\n'
 '- 있는 사람을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000444',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
