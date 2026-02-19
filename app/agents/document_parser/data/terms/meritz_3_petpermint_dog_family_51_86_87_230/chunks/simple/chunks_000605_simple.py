from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 다음의 경우에는 제1항의 절차를 대행하지 않습 니다.\n'
 '① 피보험자가 피해자에 대하여 부담하는 법률상의 손해 배상책임액이 보험증권에 기재된 보상한도액을 명백하 게 초과하는 때 ② 피보험자가 '
 '정당한 이유없이 협력하지 않은 때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 180},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000605',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
