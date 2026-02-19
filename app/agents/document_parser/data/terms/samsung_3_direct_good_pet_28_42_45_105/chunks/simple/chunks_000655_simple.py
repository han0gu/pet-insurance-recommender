from langchain_core.documents import Document

chunk = Document(
    page_content=('101 / 181\n'
 '약의 청약일은 유사계약의 청약일로 봅니다.\n'
 '<유의사항>\n'
 '최초 계약 청약일부터 5년이내 재진단 또는 치료를 받고 회사에 보험금을 청구하지 않은 경우도 재진단 또는 치료를 받은 것으로 '
 '간주합니다.\n'
 '⑤ 제4항의 재진단 또는 치료를 받지 않은 경우는 다음 각 호의 경우를 포함합니다.\n'
 '1. 검진결과 추가검사 또는 치료가 필요하지 않았던 경우 2. 제1항 제1호에서 정한 특정신체부위에 발생한 질병 또는 제1항 제2호에서 '
 '정한 특 정질병이 악화되지 않고 유지된 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
