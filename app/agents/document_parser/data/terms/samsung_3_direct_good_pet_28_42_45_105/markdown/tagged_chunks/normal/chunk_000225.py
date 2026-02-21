from langchain_core.documents import Document

chunk = Document(
    page_content=('을 중단할 것을 요구하는 경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1항\n'
 '에 따른 납입최고(독촉) 등을 실시할 것\n'
 '4. 전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것\n'
 '5. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것- \n'
 '⑦ 제1항에 따라 계약이 해지된 경우에는 제35조(해약환급금)에서 정한 해약환급금을 계'),
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
 'indexing': {'chunk_id': 'chunk_000225',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
