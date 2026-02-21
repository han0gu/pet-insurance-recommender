from langchain_core.documents import Document

chunk = Document(
    page_content=('서면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.\n'
 '1. 보험종목질- 2. 보험기간\n'
 '- 3. 보험료 납입방법 및 납입기간\n'
 '- 4. 계약자, 피보험자\n'
 '- 5. 보험가입금액(배상책임의 경우 보상한도액) 등 기타 계약의 내용\n'
 '- 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이상 지난 유효한 계약으로\n'
 '\uf000- 서 그 보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에\n'
 '- 따라 이를 변경하여 드립니다. 제\n'
 '- \uf000 회사는 계약자가 제1항 제5호에 따라 보험가입금액 또는 보상한도액을 감액하고 도'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000504',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
