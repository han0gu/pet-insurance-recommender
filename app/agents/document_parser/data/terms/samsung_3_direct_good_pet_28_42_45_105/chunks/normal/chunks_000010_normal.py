from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자가 보험기간 중에 상해로 장해분류표([별표2] 참조. 이하 같습니다)에서 정한 장해지급률이 80% 이상에 해당하는 '
 '장해상태가 되었을 때에는 최초 1회에 한하여 보험증권에 기재된 보험가입금액을 상해 후유장해(80%이상) 보험금으로 보험수익자에게 '
 '지급합니다.\n'
 '<용어풀이>\n'
 '[장해지급률]\n'
 '질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로 나타낸 것을 말합니다.\n'
 '제4조 (보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 29},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
