from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[습관성 유산, 불임 및 인공수정 관련 합병증] 한국표준질병·사인분류상의 N96~N98에 해당하는 질병을 말합니다.\n'
 'ㅜ!ㅜ!! OI 그의 〔 거레 I I ニ」 CH I IIコー I I\n'
 '열거된 행위로 인하여 각 특별약관별 보험금의 지급사유의 보험금 지급사유가 발생한 때에는 해당 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 46},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
