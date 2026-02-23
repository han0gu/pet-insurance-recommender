from langchain_core.documents import Document

chunk = Document(
    page_content=('는 녹취 등을 통해 확인받아야 하며, 설명서를 제공하여야 합니다.\n'
 '② 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제공 사실에 관하여 계약자와 회\n'
 '사간에 다툼이 있는 경우에는 회사가 이를 증명하여야 합니다.\n'
 '③ 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료의 내용이 약관의 내\n'
 '용과 다른 경우에는 계약자에게 유리한 내용으로 계약이 성립된 것으로 봅니다.- \n'
 '<용어풀이># [보험안내자료]계약의 청약을 권유하기 위해 만든 자료 등을 말합니다.\n'
 '[기명날인]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000261',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
