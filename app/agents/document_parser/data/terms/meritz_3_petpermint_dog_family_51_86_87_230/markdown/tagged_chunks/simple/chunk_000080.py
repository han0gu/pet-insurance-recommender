from langchain_core.documents import Document

chunk = Document(
    page_content=('기간에 대하여 회사는 보험계약대출이율을 연단위 복리로\n'
 '계산한 금액을 더하여 돌려 드립니다.- ① 타인의 사망을 보험금 지급사유로 하는 계약에서 계약\n'
 '- 을 체결할 때까지 피보험자의 서면(「전자서명법」 제\n'
 '- 2조제2호에 따른 전자서명이 있는 경우로서 상법 시행\n'
 '- 령에 정하는 바에 따라 본인 확인 및 위조·변조 방지\n'
 '- 에 대한 신뢰성을 갖춘 전자문서를 포함)에 의한 동의\n'
 '- 를 얻지 않은 경우. 다만, 단체가 규약에 따라 구성원\n'
 '- 의 전부 또는 일부를 피보험자로 하는 계약을 체결하'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
