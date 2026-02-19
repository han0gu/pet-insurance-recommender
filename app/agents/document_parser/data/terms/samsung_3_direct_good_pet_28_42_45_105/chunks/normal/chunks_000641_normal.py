from langchain_core.documents import Document

chunk = Document(
    page_content=('이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.\n'
 '- 100 -\n'
 '100 / 181\n'
 '제 1조 (계약의 체결 및 효력)\n'
 '① 이 특별약관은 보험계약(특별약관이 부가된 경우에는 특별약관을 포함합니다. 이하 「보험계약」이라 합니다)을 체결 또는 변경할 때 다음 '
 '각 호의 경우 보험계약자(이하 「계약자」라 합니다)의 청약과 보험회사(이하「회사」라 합니다)의 승낙으로 계약에 부가하여 이루어집니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000641',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
