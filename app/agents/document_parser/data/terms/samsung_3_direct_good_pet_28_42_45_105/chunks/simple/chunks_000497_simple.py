from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하「의료비」라 합니다)을 제5항에 따라 보험가입금액을 한도로 보 험수익자에게 반려견 의료비(치과및구강질환포함) '
 '보험금(수술당일)(이하 「의료비보 험금(수술당일)」라 합니다)으로 보상하여 드립니다. ② 반려견이 제1항의 사고로 치료를 받던 중에 '
 '보험기간이 만료된 경우에도 만료일부터 90일 이내의 의료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 180일이내 의 치료인 '
 '경우에 한합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 81},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000497',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
