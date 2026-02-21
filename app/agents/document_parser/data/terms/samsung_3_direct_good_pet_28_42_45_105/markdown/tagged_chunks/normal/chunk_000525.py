from langchain_core.documents import Document

chunk = Document(
    page_content=('- 6. 수탁기관 위탁비용 영수증 및 동물관리위탁업자가 제공하는 계약서(위탁관리업소\n'
 '- 등록번호, 업소명 및 주소, 전화번호, 위탁관리동물 종류, 품종, 나이, 서비스 기간,\n'
 '- 비용 등 포함)\n'
 '- 93 -# 원 또는 국외의 의료관련법에서 정한 의료기관에서 발급한 것이어야 합니다.# <관련법규># [의료법 제3조(의료기관)]이 '
 '법에서 의료기관이라 함은 의료인이 공중 또는 특정 다수인을 위하여 의료·조산의 업을 행하는\n'
 '곳을 말합니다. 의료기관은 종합병원·병원·치과병원·한방병원·요양병원·정신병원·의원·치과의원·한'),
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
 'indexing': {'chunk_id': 'chunk_000525',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
