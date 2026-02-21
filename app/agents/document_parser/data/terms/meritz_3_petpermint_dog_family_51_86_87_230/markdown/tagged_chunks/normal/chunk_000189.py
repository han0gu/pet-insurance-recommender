from langchain_core.documents import Document

chunk = Document(
    page_content=('하였을 때 이미 계약나이에 도달한 경우에는 유효한 계약으\n'
 '로 봅니다.\n'
 '\uf000 회사의 고의 또는 과실로 계약이 무효로 된 경우와 회사\n'
 '가 승낙 전에 무효임을 알았거나 알 수 있었음에도 보험료\n'
 '를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날부\n'
 '터 반환일까지의 기간에 대하여 회사는 보험계약대출이율을\n'
 '연단위 복리로 계산한 금액을 더하여 돌려 드립니다.# 제13조(계약내용의 변경 등)\uf000 계약자는 회사의 승낙을 얻어 다음의 사항을 '
 '변경할 수\n'
 '있습니다. 이 경우 승낙을 서면 등으로 알리거나 보험증권'),
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
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
