from langchain_core.documents import Document

chunk = Document(
    page_content=('제23조 (특별약관 내용의 변경 등)\n'
 '① 회사는 계약자가 보험기간 중 회사의 승낙을 얻어 기본계약의 내용을 변경할 때 동일 내용으로 변경합니다. 이 경우 승낙을 서면 등으로 '
 '알리거나 보험증권의 뒷면에 기재 하여 드립니다. ② 계약자는 보험수익자를 변경할 수 있으며 이 경우에는 회사의 승낙이 필요하지 않습 '
 '니다. 다만, 변경된 보험수익자가 회사에 권리를 대항하기 위해서는 계약자가 보험수 익자가 변경되었음을 회사에 통지하여야 합니다.\n'
 '<유의사항>\n'
 '계약자가 회사에 보험수익자가 변경되었음을 통지하기 전에 보험금 지급사유가 발생한 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 53},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000236',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
