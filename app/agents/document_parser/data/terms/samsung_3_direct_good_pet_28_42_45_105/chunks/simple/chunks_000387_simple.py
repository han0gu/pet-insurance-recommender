from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 보통약관 제21조(계약 내용의 변경 등)에 따라 계약내용을 변경할 수 '
 '있습니다.\n'
 '<유의사항>\n'
 '[위험변경에 따른 계약변경 절차]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 71},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000387',
              'chunk_char_len': 106,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
