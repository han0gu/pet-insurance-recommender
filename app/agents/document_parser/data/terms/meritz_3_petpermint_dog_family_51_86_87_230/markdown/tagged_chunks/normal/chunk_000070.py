from langchain_core.documents import Document

chunk = Document(
    page_content=('에 준하는 전자적 의사표시(이하‘서면 등’이라 합니다)를\n'
 '발송한 때 효력이 발생합니다. 계약자는 서면 등을 발송한\n'
 '때에 그 발송 사실을 회사에 지체없이 알려야 합니다.\n'
 '\uf000 계약자가 청약을 철회한 때에는 회사는 청약의 철회를\n'
 '접수한 날부터 3영업일 이내에 납입한 보험료를 돌려드리68며, 보험료 반환이 늦어진 기간에 대하여는 이 계약의 보험\n'
 '계약대출이율을 연단위 복리로 계산한 금액을 더하여 지급\n'
 '합니다. 다만, 계약자가 제1회 보험료를 신용카드로 납입한\n'
 '계약의 청약을 철회하는 경우에는 회사는 청약의 철회를 접'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
