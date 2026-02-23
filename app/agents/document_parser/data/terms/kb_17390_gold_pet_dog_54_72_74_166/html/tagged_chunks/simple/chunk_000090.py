from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 상법 제651조의2(서면에 의한 질문의 효력) 보험자가 서면으로 질문한 사항은 중요한 사항으로 '
 "추정한다.</td></tr></tbody></table><br><table id='110' "
 "style='font-size:16px'><thead></thead><tbody><tr><td></td></tr><tr><td>관 련 법 "
 '규 의료법 ∙ 의료법 제3조(의료기관) 내지 제3조의2(병원등)의 규정에 의한 병원 공 30개 이상의 병상(또는 요양병상)을 갖춘 병원, '
 '치과병원, 한방병원(또는 요양병 통 원) 사항 ∙ 의료법'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
