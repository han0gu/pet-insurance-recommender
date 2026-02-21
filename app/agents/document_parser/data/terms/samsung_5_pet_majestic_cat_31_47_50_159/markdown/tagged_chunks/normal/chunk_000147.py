from langchain_core.documents import Document

chunk = Document(
    page_content=('- 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보험회사 및 보험관련\n'
 '- 단체 등에 개인정보를 제공할 수 있습니다.\n'
 '- ② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.\n'
 '# 제46조 (준거법)이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 ｢금융소\n'
 '비자 보호에 관한 법률｣, 상법, 민법 등 관계 법령을 따릅니다.제 47조 (예금보험에 의한 지급보장)회사가 파산 등으로 인하여 보험금 '
 '등을 지급하지 못할 경우에는 예금자보호법에서 정하'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000147',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
