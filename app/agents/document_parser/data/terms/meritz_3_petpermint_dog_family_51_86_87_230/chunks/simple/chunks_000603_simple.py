from langchain_core.documents import Document

chunk = Document(
    page_content=('【손해배상청구에 대한 회사의 해결 예시】\n'
 '※ 항변이란 어떤 일을 부당하다고 여겨 따지거나 반대하 는 뜻을 밝힌다는 것을 의미합니다.\n'
 '제10조(합의․절충․중재․소송의 협조․대행 등)\n'
 '\uf000 회사는 피보험자의 법률상 손해배상책임을 확정하기 위 하여 피보험자가 피해자와 행하는 합의·절충·중재 또는 소송(확인의 소를 '
 '포함합니다)에 대하여 협조하거나, 피보 험자를 위하여 이러한 절차를 대행할 수 있습니다. \uf000 회사는 피보험자에 대하여 보상책임을 '
 '지는 한도 내에서 제1항의 절차에 협조하거나 대행합니다.\n'
 '【보상책임을 지는 한도】'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 180},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000603',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
