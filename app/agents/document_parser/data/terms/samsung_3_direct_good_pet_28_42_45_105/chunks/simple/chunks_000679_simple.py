from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조 (특별약관의 보험기간)\n'
 '① 이 특별약관의 보험기간은 보험계약의 보험기간 내에서 회사가 정한 기간으로 합니다. ② 이 특별약관의 보험료는 보험계약의 납입기간 중에 '
 '보험계약의 보험료와 함께 납입하 여야 하며, 보험계약의 보험료를 선납하는 경우에도 또한 같습니다.\n'
 '제5조 (특별약관 내용의 변경)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000679',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
