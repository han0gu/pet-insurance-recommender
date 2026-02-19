from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험가입금액 제한]\n'
 '피보험자가 가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 말합니다.\n'
 '[일부보장 제외] 일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 특정 질병 또는 특정 신 체 부위를 보장에서 '
 '제외하는 방법을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000241',
              'chunk_char_len': 151,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
