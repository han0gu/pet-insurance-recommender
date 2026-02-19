from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험증권에 기재된 피보험자가 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중에 진단확정된 질병으로 병원 또는 '
 '의원(한방병원 또는 한의원을 포함합니 다)에 1일이상 계속 입원하여 치료를 받은 경우에는 입원기간 동안 보험증권에 기재 된 반려견을 '
 '수탁기관에 위탁함으로써 발생한 위탁비용을 반려견 위탁비용으로 보험 수익자에게 지급합니다. 다만, 반려견 위탁비용의 지급일수는 1회 입원당 '
 '180일을 한 도로 하며, 피보험자의 입원기간을 초과할 수 없습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000599',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
