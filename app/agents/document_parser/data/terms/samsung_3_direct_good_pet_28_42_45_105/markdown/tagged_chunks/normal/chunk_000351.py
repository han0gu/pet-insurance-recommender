from langchain_core.documents import Document

chunk = Document(
    page_content=('수 있었음에도 불구하고 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날\n'
 '부터 반환일까지의 기간에 대하여 회사는 이 특별약관의 보험계약대출이율을 연단위 복\n'
 '리로 계산한 금액을 더하여 돌려 드립니다.# 제 17조 (특별약관 내용의 변경 등)① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 '
 '수 있습니다. 이 경우 승낙을 서\n'
 '면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.- 1. 보험종목\n'
 '- 2. 보험기간\n'
 '- 3. 보험료 납입주기, 납입방법 및 납입기간\n'
 '- 4. 계약자, 피보험자, 반려견'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000351',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
