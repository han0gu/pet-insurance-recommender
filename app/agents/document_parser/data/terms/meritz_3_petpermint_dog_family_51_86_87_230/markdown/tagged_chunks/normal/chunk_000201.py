from langchain_core.documents import Document

chunk = Document(
    page_content=('안내가 계약자에게 도달하지 못한 경우 포함)에는 직전계약\n'
 '과 동일한 조건으로 보험계약을 연장합니다. 다만, 보험계\n'
 '약이 연장된 경우 연장된 날 기준으로 매년 현재의 예정기102초율(적용이율, 적용위험률, 부가보험요율) 적용 및 반려동\n'
 '물의 연령 증가 등의 사유로 보험요율이 변동될 수 있으며\n'
 '이 때의 보험료는「보험료 및 해약환급금 산출방법서」에\n'
 '따라 산출합니다.\uf000 제6항에 따라 보험계약이 연장된 경우 계약자는 그 최초\n'
 '연장된 날로부터 90일 이내에 그 계약을 취소할 수 있으며,\n'
 '계약자가 연장된 보험계약을 취소하는 경우 회사는 최초연'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000201',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
