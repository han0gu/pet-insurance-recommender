from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 회사는 계약자에게 갱신전 계약의 보험기간이 끝나기 15일 이전까지 갱신 요건, 보장 내용 변경내역, 갱신보험료 및 갱신 절차 등을 '
 '서면(등기우편 등), 전화(음성녹취) 또 는 전자문서(SMS 포함) 등으로 알려 드립니다.(단, 갱신보험료 납입면제가 발생한 경 우는 '
 '제외합니다.)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000810',
              'chunk_char_len': 159,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
