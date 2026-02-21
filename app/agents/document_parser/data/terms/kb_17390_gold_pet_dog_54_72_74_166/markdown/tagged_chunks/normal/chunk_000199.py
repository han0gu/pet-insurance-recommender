from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라도 회사는 이를 지급하지 않습니다.\n'
 '등제8관 분쟁의 조정# 제43조(분쟁의\n'
 '\uf000 계약에조정)관하여 분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금- 융감독원장에게 조정을 신청할 수 있으며, '
 '분쟁조정 과정에서 계약자는 관계 법령\n'
 '- 이 정하는 바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본의 제공 또\n'
 '- 는 청취를 포함한다)을 요구할 수 있습니다.\n'
 '- \uf000 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
