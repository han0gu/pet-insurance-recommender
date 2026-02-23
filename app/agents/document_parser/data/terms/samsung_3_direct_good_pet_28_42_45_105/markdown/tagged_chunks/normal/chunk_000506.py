from langchain_core.documents import Document

chunk = Document(
    page_content=('수술당일제외, 검사비포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사\n'
 '항을 따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다.- 90 -# 3-6. [갱신형] 반려견 '
 '위탁비용(반려인상해입원1 일이상180일한도)(실손) 특별약관# 제1조 (보험금의 지급사유)- ① 회사는 보험증권에 기재된 피보험자가 이 '
 '특별약관의 보험기간(이하 「보험기간」 이라\n'
 '- 합니다) 중에 상해를 입고 그 직접결과로써 생활기능 또는 업무능력에 지장을 가져와'),
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
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000506',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
