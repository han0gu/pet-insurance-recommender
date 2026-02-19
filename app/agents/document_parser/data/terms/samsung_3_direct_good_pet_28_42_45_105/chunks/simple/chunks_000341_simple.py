from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하「의료비」라 합니다)을 제5항에 따라 보험가입금액 을 한도로 보험수익자에게 반려견 의료비(치과및구강질환포함) 보험금(수술당일제외, '
 '검사비포함)(이하 「의료비보험금(수술당일제외, 검사비포함)」라 합니다)으로 보상하 여 드립니다. 단, 보험기간 중에 발생한 사고로 회사가 '
 '지급하는 연간 의료비보험금( 수술당일제외, 검사비포함)의 총 합계는 보험증권에 기재된 연간 총 보상한도액을 한 도로 합니다. ② 반려견이 '
 '제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터 90일 이내의 의료비는 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 67},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000341',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
