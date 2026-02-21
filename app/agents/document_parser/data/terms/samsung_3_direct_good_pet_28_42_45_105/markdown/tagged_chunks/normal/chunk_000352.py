from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 보험료 납입주기, 납입방법 및 납입기간\n'
 '- 4. 계약자, 피보험자, 반려견\n'
 '- 5. 보험가입금액(배상책임의 경우 보상한도액) 등 기타 계약의 내용\n'
 '- ② 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 계약으로서 그\n'
 '- 보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에 따라 이를\n'
 '- 변경하여 드립니다.\n'
 '- ③ 회사는 계약자가 제1항 제5호에 따라 보험가입금액(배상책임의 경우 보상한도액)을\n'
 '- 감액하고자 할 때에는 그 감액된 부분은 특별약관이 해지된 것으로 보며, 이로써 회사'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000352',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
