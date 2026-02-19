from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항에 따라 정상적으로 갱신이 이루어진 경우 갱신계약의 보장은 갱신전 계약에 의 한 보장이 끝나는 때부터 적용합니다. ③ 제1항에도 '
 '불구하고 갱신전 계약에서 소멸사유가 발생한 경우에는 해당 갱신형 계약 은 갱신되지 않습니다. ④ 제3항에도 불구하고 보험금 청구 지연 '
 '등의 사유로 갱신이 이루어진 경우에는 해당 갱신계약을 무효로 하며 소멸사유 발생 이후 납입한 해당 보험료를 돌려 드립니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000809',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
