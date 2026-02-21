from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '을 의미합니다.제14조(합의․절충․중재․소송의 협조․대행 등)\n'
 '\uf000 회사는 피보험자의 법률상 손해배상책임을 확정하기 위하여 피보험자가 피해자와\n'
 '행하는 합의·절충·중재 또는 소송(확인의 소를 포함합니다)에 대하여 협조하거나, 피보험자를 위하여 이러한 절차를 대행할 수 있습니다.- '
 '\uf000 회사는 피보험자에 대하여 보상책임을 지는 한도 내에서 제1항의 절차에 협조하 질\n'
 '- 거나 대행합니다. 병\n'
 '- \uf000 회사가 제1항의 절차에 협조하거나 대행하는 경우에는 피보험자는 회사의 요청에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000689',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
