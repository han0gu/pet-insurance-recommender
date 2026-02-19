from langchain_core.documents import Document

chunk = Document(
    page_content=('제16조(타인을 위한 계약)\n'
 '\uf000 계약자는 타인을 위한 계약을 체결하는 경우에 그 타인 의 위임이 없는 때에는 반드시 이를 회사에 알려야 하며, 이를 알리지 '
 '않았을 때에는 그 타인은 이 계약이 체결된 사 실을 알지 못하였다는 사유로 회사에 이의를 제기 할 수 없 습니다. \uf000 타인을 '
 '위한 계약에서 보험사고가 발생한 경우에 계약자 가 그 타인에게 보험사고의 발생으로 생긴 손해를 배상한 때에는 계약자는 그 타인의 권리를 '
 '해하지 않는 범위 안에 서 회사에 보험금의 지급을 청구할 수 있습니다.\n'
 '제17조(준용규정)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 184},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000627',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
