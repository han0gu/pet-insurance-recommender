from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제15조(계약 전 알릴 의무)에 따라 계약자 또는 피보 험자가 회사에 알린 내용이나 건강진단 내용이 보험금 지급사유의 발생에 영향을 '
 '미쳤음을 회사가 증명하는 경우 ② 제17조(알릴 의무 위반의 효과)를 준용하여 회사가 보 장을 하지 않을 수 있는 경우 ③ 진단계약에서 '
 '보험금 지급사유가 발생할 때까지 진단 을 받지 않은 경우. 다만, 진단계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 '
 '발생하는 경 우에는 보장을 해드립니다.\n'
 '제27조(제2회 이후 보험료의 납입)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 75},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
