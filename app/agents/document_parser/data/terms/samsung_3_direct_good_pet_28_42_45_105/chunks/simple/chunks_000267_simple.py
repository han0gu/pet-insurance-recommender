from langchain_core.documents import Document

chunk = Document(
    page_content=('55 / 181\n'
 '관 내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 보험수익자로 변경하여 특별 약관의 특별부활(효력회복)을 청약할 수 있음을 '
 '보험수익자에게 통지하여야 합니다.\n'
 '<용어풀이>\n'
 '[강제집행과 담보권실행]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 56},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000267',
              'chunk_char_len': 121,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
