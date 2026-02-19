from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하「의료비」라 합니다)에서「자기부담금」및「반 려견의료비(치과및구강질환포함)(수술당일제외,검사비 포함)보험금의 1일 한도 」를 제외한 '
 '금액을 아래에 정한 한도로 제5항에 따라 보험수익자에게 반려견의료비확대보 장 보험금으로 보상하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000456',
              'chunk_char_len': 136,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
