from langchain_core.documents import Document

chunk = Document(
    page_content=('기일」이라 합니다)이 지나도록 보험금을 지급하지 않았을\n'
 '때에는 지급기일의 다음날부터 지급일까지의 기간에 대하여【별표1(보험금을 지급할 때의 적립이율 계산)】에서 정한\n'
 '이율로 계산한 금액을 보험금에 더하여 지급합니다. 그러나\n'
 '계약자 또는 피보험자의 책임있는 사유로 지급이 지연된 때\n'
 '에는 그 해당기간에 대한 이자는 더하여 지급하지 않습니\n'
 '다. 다만, 회사는 계약자 등이 분쟁조정을 신청했다는 사유\n'
 '만으로 이자지급을 거절하지 않습니다.177# 【가지급보험금】보험금이 지급기한 내에 지급되지 못할 것으로 판단되는'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000491',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
