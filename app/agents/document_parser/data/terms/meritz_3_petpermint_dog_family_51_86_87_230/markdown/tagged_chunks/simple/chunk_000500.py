from langchain_core.documents import Document

chunk = Document(
    page_content=('항변으로써 대항가능※ 항변이란 어떤 일을 부당하다고 여겨 따지거나 반대하\n'
 '는 뜻을 밝힌다는 것을 의미합니다.# 제10조(합의․절충․중재․소송의 협조․대행 등)\uf000 회사는 피보험자의 법률상 손해배상책임을 '
 '확정하기 위\n'
 '하여 피보험자가 피해자와 행하는 합의·절충·중재 또는\n'
 '소송(확인의 소를 포함합니다)에 대하여 협조하거나, 피보\n'
 '험자를 위하여 이러한 절차를 대행할 수 있습니다.\n'
 '\uf000 회사는 피보험자에 대하여 보상책임을 지는 한도 내에서'),
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
 'indexing': {'chunk_id': 'chunk_000500',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
