from langchain_core.documents import Document

chunk = Document(
    page_content=('용, 조회 또는 제공하지 않습니다. 다만, 회사는 이 특별약관의 체결, 유지, 보험금 지\n'
 '급 등을 위하여 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보험회\n'
 '사 및 보험관련단체 등에 개인정보를 제공할 수 있습니다.\n'
 '② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.\n'
 '제45조 (준거법)\n'
 '이 특별약관은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 ｢금 융소비자 보호에 관한 법률｣, 상법, 민법 등 '
 '관계 법령을 따릅니다.\n'
 '제 46조 (예금보험에 의한 지급보장)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 64},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000310',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
