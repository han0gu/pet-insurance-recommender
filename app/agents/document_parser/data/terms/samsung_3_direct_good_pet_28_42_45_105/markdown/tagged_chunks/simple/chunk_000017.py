from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보험금 지급사유와 보험계약일로부터 2년이 지난 후에 발생한 습관성 유산, 불임\n'
 '- 및 인공수정 관련 합병증으로 인한 경우에는 보험금을 지급합니다.\n'
 '<용어풀이>[습관성 유산, 불임 및 인공수정 관련 합병증]\n'
 '한국표준질병·사인분류상의 N96~N98에 해당하는 질병을 말합니다.5. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동- 29 -# '
 '한 때에는 해당 보험금을 지급하지 않습니다.- 1. 전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙벽을 오르내리거나 특수한 기'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
