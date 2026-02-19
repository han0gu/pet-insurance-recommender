from langchain_core.documents import Document

chunk = Document(
    page_content=('제18조 (사기에 의한 계약)\n'
 '계약자 또는 피보험자가 대리진단, 약물사용을 수단으로 진단절차를 통과하거나 진단서 위·변조 또는 청약일 이전에 암 또는 '
 '사람면역결핍바이러스(HIV) 감염의 진단확정을 받은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우 '
 '에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니 다.\n'
 '제4관 보험계약의 성립과 유지'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 229,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
