from langchain_core.documents import Document

chunk = Document(
    page_content=('반드시 사실대로 알려야(이하 "계약 전<br>알릴 의무"라 하며, 상법상 "고지의무"와 같습니다)합니다.</p><br><table '
 "id='4' style='font-size:14px'><thead></thead><tbody><tr><td>관 련 법</td><td>규 "
 '상법</td></tr><tr><td colspan="2">∙ 상법 제651조(고지의무위반으로 인한 계약해지) 보험계약당시에 보험계약자 '
 '또는 피보험자가 고의 또는 중대한 과실로 인하 여 중 요한 사항을 고지하지 아니하거나 부실의 고지를 한 때에는 보험자는 그 사실을 안'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001207',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
