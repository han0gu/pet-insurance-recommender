from langchain_core.documents import Document

chunk = Document(
    page_content=('- 단기준은 회사에서 정한 계약사정기준을 따르며, 개개인의 질병의 상태 등에 대한 의\n'
 '- 사의 소견에 따라서 다르게 적용할 수 있습니다.\n'
 '- ③ 제1항 제1호 및 제2호의 규정에 의하여 보험계약에 부가된 조건을 보험증권의 뒷면에\n'
 '- 기재하여 드립니다.\n'
 '# 제4조 (특별약관의 보험기간)- ① 이 특별약관의 보험기간은 보험계약의 보험기간 내에서 회사가 정한 기간으로 합니다.\n'
 '- ② 이 특별약관의 보험료는 보험계약의 납입기간 중에 보험계약의 보험료와 함께 납입하\n'
 '- 여야 하며, 보험계약의 보험료를 선납하는 경우에도 또한 같습니다.'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000588',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
