from langchain_core.documents import Document

chunk = Document(
    page_content=('72 / 181\n'
 '이자를 더하여 지급하지 않습니다.\n'
 '제 15조 (사기에 의한 계약)\n'
 '계약자 또는 피보험자의 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에는 계약체결일부터 5년 이내(사기사실을 안 날부터 1개월 '
 '이내)에 계약을 취소할 수 있습니 다.\n'
 '제 16조 (특별약관의 무효)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000405',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
