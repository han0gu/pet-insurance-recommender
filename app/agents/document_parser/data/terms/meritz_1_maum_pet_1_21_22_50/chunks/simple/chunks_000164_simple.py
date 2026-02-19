from langchain_core.documents import Document

chunk = Document(
    page_content=('제13조(합의․절충․중재․소송의 협조․대행 등)\n'
 '① 회사는 피보험자의 법률상 손해배상책임을 확정하기 위하여 피보험자가 피해자와 행하 는 합의·절충·중재 또는 소송(확인의 소를 '
 '포함합니다)에 대하여 협조하거나, 피보험자 를 위하여 이러한 절차를 대행할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 147,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
