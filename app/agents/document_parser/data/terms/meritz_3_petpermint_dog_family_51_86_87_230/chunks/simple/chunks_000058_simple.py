from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경 우에는 보통약관 제23조(계약내용의 변경 등)에 따라 계약 내용을 '
 '변경할 수 있습니다.\n'
 '\uf000 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소 된 경우에는 보험료를 감액하고, 이후 기간 보장을 위한 재 원인 '
 '계약자적립액 등의 차이로 인하여 발생한 정산금액(이'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 63},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000058',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
