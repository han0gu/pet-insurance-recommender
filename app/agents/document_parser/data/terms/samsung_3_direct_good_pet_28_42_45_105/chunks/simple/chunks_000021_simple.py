from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[습관성 유산, 불임 및 인공수정 관련 합병증] 한국표준질병·사인분류상의 N96~N98에 해당하는 질병을 말합니다.\n'
 '5. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 29},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000021',
              'chunk_char_len': 103,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
