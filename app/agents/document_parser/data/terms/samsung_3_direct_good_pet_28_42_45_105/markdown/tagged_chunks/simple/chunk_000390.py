from langchain_core.documents import Document

chunk = Document(
    page_content=('- 환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '# 제 28조 (준용규정)이 특별약관에 정하지 않은 사항은 특별약관 일반사항을 따르며, 이 특별약관 및 특별약관 일반사항에 정하지 않은 '
 '사항은 보통약관을 따릅니다.- 76 -76 / 1813-2. 반려견 의료비 확대보장 (특정처치(이물제거))(수술당일제외,\n'
 '연간2회한)(재가입형) 특별약관# 제 1조 (보험금의 지급사유)① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 '
 '합니다) 중'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000390',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
