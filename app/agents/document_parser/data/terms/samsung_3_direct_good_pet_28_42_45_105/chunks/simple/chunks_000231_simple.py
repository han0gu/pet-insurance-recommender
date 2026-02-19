from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 이 특별약관의 '
 '보험계약대출이율을 연단위 복 리로 계산한 금액을 더하여 지급합니다.\n'
 '제 22조 (특별약관의 무효)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 52},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000231',
              'chunk_char_len': 131,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
