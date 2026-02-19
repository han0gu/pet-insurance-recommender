from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조(보험료정산기간)\n'
 '① 계약자와 회사는 보험증권에 정한 기간(이하 「정산기간」 이라 합니다)마다 보험료를 정산하기로 합 니다. ② 정산기간은 다음 중 어느 '
 '하나를 정하여 보험증권에 기재합니다.\n'
 '- 매월, 매분기, 매반기, 기타() - 보험기간 종료 후\n'
 '제3조(예치보험료)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 38},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000190',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
